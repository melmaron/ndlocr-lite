GUI_DEFAULT_CONFIG = {
    'langcode': 'ja',
    'json': True,
    'xml': True,
    'tei': True,
    'txt': True,
    'markdown': False,
    'pdf': False,
    'pdf_viztxt': False,
    'pdf_merge': True,
    'pdf_page_output': True,
    'pdf_resolution': 300,
    'text_daisy': False,
    'result_markdown_display': False,
    'selected_output_path': None,
    'visualize': True,
    'device': 'auto',
    'device_id': 0,
    'batch_size': 0,
    'precision': 'fp32',
}


def build_default_config():
    return dict(GUI_DEFAULT_CONFIG)
