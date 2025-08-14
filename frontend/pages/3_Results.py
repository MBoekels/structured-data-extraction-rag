# pages/3_Results.py
import streamlit as st
import sqlite3
import json
import os
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from settings import API_BASE, DB_PATH

st.title("📈 Results Overview")

# No longer need results_ui_state as we are not using expanders for each result
# if 'results_ui_state' not in st.session_state:
#     st.session_state.results_ui_state = {}

@st.cache_data(ttl=30)
def fetch_results(query_id=None, company_id=None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = """
            SELECT
                c.name AS company_name,
                q.query_text AS query_text,
                r.llm_output as answer,
                r.source_chunks as source,
                r.created_at,
                r.query_id,
                r.company_id
            FROM results r
            JOIN companies c ON r.company_id = c.id
            JOIN queries q ON r.query_id = q.id
        """
        
        conditions = []
        params = []
        
        if query_id:
            conditions.append("r.query_id = ?")
            params.append(query_id)
        
        if company_id:
            conditions.append("r.company_id = ?")
            params.append(company_id)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY r.created_at DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

@st.cache_data
def get_pdf_path(pdf_id):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM pdfs WHERE id = ?", (pdf_id,))
        row = cursor.fetchone()
        return row[0] if row else None

@st.cache_data
def fetch_queries_for_filter():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, query_text FROM queries")
        return [dict(row) for row in cursor.fetchall()]

@st.cache_data
def fetch_companies_for_filter():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM companies")
        return [dict(row) for row in cursor.fetchall()]

def render_pdf_page_with_bboxes(file_path, page_num, bboxes, encompass=False):
    try:
        pdf = pdfium.PdfDocument(file_path)
        page = pdf.get_page(page_num - 1)  # pypdfium2 is 0-indexed
        image = page.render().to_pil()

        draw = ImageDraw.Draw(image)
        page_width, page_height = image.size

        if encompass and bboxes:
            min_x = min(b['l'] for b in bboxes) / 1000 * page_width
            min_y = min(b['b'] for b in bboxes) / 1000 * page_height
            max_x = max(b['r'] for b in bboxes) / 1000 * page_width
            max_y = max(b['t'] for b in bboxes) / 1000 * page_height
            draw.rectangle([min_x, min_y, max_x, max_y], outline="green", width=3)
        else:
            for bbox_dict in bboxes:
                # The bbox format from docling is a dictionary with keys l, t, r, b
                x0 = bbox_dict['l']
                y0 = bbox_dict['b'] # Invert t and b for correct drawing
                x1 = bbox_dict['r']
                y1 = bbox_dict['t']
                
                # Convert bbox coordinates from relative to absolute
                abs_bbox = [
                    x0 / 1000 * page_width,
                    y0 / 1000 * page_height,
                    x1 / 1000 * page_width,
                    y1 / 1000 * page_height,
                ]
                draw.rectangle(abs_bbox, outline="red", width=2)

        return image
    except Exception as e:
        st.error(f"Failed to render PDF page: {e}")
        return None

def crop_image_to_bboxes(image, bboxes):
    if not bboxes:
        return image

    page_width, page_height = image.size

    # Find the min and max coordinates to create an encompassing bounding box
    min_x = min(b['l'] for b in bboxes) / 1000 * page_width
    min_y = min(b['b'] for b in bboxes) / 1000 * page_height # Invert t and b for correct cropping
    max_x = max(b['r'] for b in bboxes) / 1000 * page_width
    max_y = max(b['t'] for b in bboxes) / 1000 * page_height

    # Add some padding
    padding = 50
    crop_box = (
        max(0, min_x - padding),
        max(0, min_y - padding),
        min(page_width, max_x + padding),
        min(page_height, max_y + padding),
    )

    # Ensure the crop box is not smaller than a quarter of the page
    min_crop_width = page_width / 2
    min_crop_height = page_height / 2

    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]

    if crop_width < min_crop_width:
        center_x = (crop_box[0] + crop_box[2]) / 2
        crop_box = (
            max(0, min(center_x - min_crop_width / 2, min_crop_width)),
            crop_box[1],
            min(page_width, max(center_x + min_crop_width / 2, min_crop_width)),
            crop_box[3],
        )
    
    if crop_height < min_crop_height:
        center_y = (crop_box[1] + crop_box[3]) / 2
        crop_box = (
            crop_box[0],
            max(0, min(center_y - min_crop_height / 2, min_crop_height)),
            crop_box[2],
            min(page_height, max(center_y + min_crop_height / 2, min_crop_height)),
        )

    return image.crop(crop_box)

queries = fetch_queries_for_filter()
companies = fetch_companies_for_filter()

query_options = {q['id']: f"{q['id']}: {q['query_text']}" for q in queries}
company_options = {c['id']: c['name'] for c in companies}

query_options[None] = "All Queries"
company_options[None] = "All Companies"

selected_query_id = st.selectbox("Filter by Query", options=list(query_options.keys()), format_func=lambda x: query_options[x])
selected_company_id = st.selectbox("Filter by Company", options=list(company_options.keys()), format_func=lambda x: company_options[x])

results = fetch_results(query_id=selected_query_id, company_id=selected_company_id)

if not results:
    st.info("No results found for the selected filters.")
else:
    st.write(f"### Displaying {len(results)} most recent results")

    # Create display names for the selectbox
    result_display_names = [f"{r['company_name']} - {r['created_at']} (Query: {r['query_text']})" for r in results]

    # Initialize selected_result_index in session state if not present
    if 'selected_result_index' not in st.session_state:
        st.session_state.selected_result_index = 0 # Default to the first result

    # Get the current selected index from the selectbox
    new_selected_result_index = st.selectbox(
        "Select a Result to View Details",
        options=range(len(results)),
        format_func=lambda x: result_display_names[x],
        index=st.session_state.selected_result_index # Set initial value from session state
    )

    # If the selected result changes, update session state and reset source view
    if new_selected_result_index != st.session_state.selected_result_index:
        st.session_state.selected_result_index = new_selected_result_index
        # Reset source view states for the new selection
        st.session_state.show_source = -1
        st.session_state.show_other_sources = False
        st.rerun() # Rerun to apply the state changes immediately

    # Get the currently selected result
    selected_result = results[st.session_state.selected_result_index]

    # Display the details of the selected result
    st.write("---") # Separator for clarity
    st.write(f"## Details for Selected Result")
    st.write("---")

    st.write("**Company:**", selected_result['company_name'])
    st.write("**Query:**", selected_result['query_text'])
    st.write("**Date:**", selected_result['created_at'])

    st.write("**Answer:**")
    try:
        answer_json = json.loads(selected_result['answer'])
        st.json(answer_json)
    except (json.JSONDecodeError, TypeError):
        st.text(selected_result['answer'])

    st.write("**Sources:**")
    try:
        sources = json.loads(selected_result['source'])
        if sources:
            # Display the first source by default
            first_source = sources[0]
            st.write("---")
            st.write("**Top Source (Highest Relevance):**")
            if 'llm_evaluation' in first_source:
                st.json(first_source['llm_evaluation'])

            # Use a single session state variable for show_source
            if 'show_source' not in st.session_state:
                st.session_state.show_source = -1 # -1 means no source is shown

            view_source_label = "Close Source" if st.session_state.show_source == 0 else "View Source"
            if st.button(view_source_label, key=f"view_source_selected_0"):
                st.session_state.show_source = 0 if st.session_state.show_source != 0 else -1
                st.rerun()

            if st.session_state.show_source == 0:
                # Checkbox keys are simplified as only one result is displayed at a time
                crop_view = st.checkbox("Crop to Bounding Box", key=f"crop_selected_0")
                encompass_view = st.checkbox("Encompass Bounding Boxes", value=True, key=f"encompass_selected_0")
                if 'original_metadata' in first_source and 'pdf_id' in first_source['original_metadata']:
                    pdf_id = first_source['original_metadata']['pdf_id']
                    relative_path = get_pdf_path(pdf_id)
                    if relative_path:
                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        file_path = os.path.join(project_root, 'uploads', relative_path)
                        page_num = first_source['original_metadata']['pages'][0]
                        bboxes = first_source['original_metadata']['bboxes']

                        image = render_pdf_page_with_bboxes(file_path, page_num, bboxes, encompass=encompass_view)
                        if image:
                            if crop_view:
                                image = crop_image_to_bboxes(image, bboxes)
                            st.image(image, caption=f"Page {page_num} of {file_path}", use_container_width=True)
                    else:
                        st.error(f"Could not find file path for PDF ID {pdf_id}")

            # Add a button to show other sources
            if len(sources) > 1:
                if 'show_other_sources' not in st.session_state:
                    st.session_state.show_other_sources = False

                show_other_sources_label = "Hide Other Sources" if st.session_state.show_other_sources else "Show Other Sources"
                if st.button(show_other_sources_label, key=f"show_other_sources_selected"):
                    st.session_state.show_other_sources = not st.session_state.show_other_sources
                    st.rerun()

                if st.session_state.show_other_sources:
                    for j, source in enumerate(sources[1:]):
                        st.write("---")
                        st.write(f"**Source {j + 2}:**")
                        if 'llm_evaluation' in source:
                            st.json(source['llm_evaluation'])

                        view_source_label_other = "Close Source" if st.session_state.show_source == j + 1 else "View Source"
                        if st.button(view_source_label_other, key=f"view_source_selected_{j+1}"):
                            st.session_state.show_source = j + 1 if st.session_state.show_source != j + 1 else -1
                            st.rerun()

                        if st.session_state.show_source == j + 1:
                            crop_view_other = st.checkbox("Crop to Bounding Box", key=f"crop_selected_{j+1}")
                            encompass_view_other = st.checkbox("Encompass Bounding Boxes", value=True, key=f"encompass_selected_{j+1}")
                            if 'original_metadata' in source and 'pdf_id' in source['original_metadata']:
                                pdf_id = source['original_metadata']['pdf_id']
                                relative_path = get_pdf_path(pdf_id)
                                if relative_path:
                                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                                    file_path = os.path.join(project_root, 'uploads', relative_path)
                                    page_num = source['original_metadata']['pages'][0]
                                    bboxes = source['original_metadata']['bboxes']

                                    image = render_pdf_page_with_bboxes(file_path, page_num, bboxes, encompass=encompass_view_other)
                                    if image:
                                        if crop_view_other:
                                            image = crop_image_to_bboxes(image, bboxes)
                                        st.image(image, caption=f"Page {page_num} of {file_path}", use_container_width=True)
                                else:
                                    st.error(f"Could not find file path for PDF ID {pdf_id}")
        else:
            st.info("No sources provided.")
    except (json.JSONDecodeError, TypeError) as e:
        st.error(f"Error processing sources: {e}")
        st.text(selected_result['source'])

st.markdown("---")

if st.button("Refresh Results"):
    # Reset all relevant session states when refreshing
    st.session_state.selected_result_index = 0
    st.session_state.show_source = -1
    st.session_state.show_other_sources = False
    st.rerun()