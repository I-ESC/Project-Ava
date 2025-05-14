import asyncio
import html
import io
import csv
import json
import logging
import os
import re
import time
import coloredlogs
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union, List
import xml.etree.ElementTree as ET

import numpy as np
import tiktoken

ENCODER = None

logger = logging.getLogger("videorag")
coloredlogs.install(level='INFO', logger=logger)


def set_logger(log_file: str):
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()

def clean_json(result:str):
    result = result.replace("```", "").replace("json", "").strip()
    # pattern = r"\{.*?\}"
    # pattern = r'\{\s*"Entities":\s*\[.*?\],\s*"Relations":\s*\[.*?\]\s*\}'
    # matches = re.findall(pattern, result, re.S)
    # return matches[0]
    return result

# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input
    
    result = html.unescape(input.strip().replace("/", " "))
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

def calculate_cosine_similarity(mat1, mat2):
    """
    Calculate cosine similarity between two matrices using numpy.
    
    Args:
        mat1 (np.ndarray): First matrix, shape (n1, feature_dim)
        mat2 (np.ndarray): Second matrix, shape (n2, feature_dim)
    
    Returns:
        np.ndarray: Cosine similarity matrix, shape (n1, n2)
    """
    # Normalize the matrices
    mat1_norms = np.linalg.norm(mat1, axis=1, keepdims=True)  # Shape (n1, 1)
    mat2_norms = np.linalg.norm(mat2, axis=1, keepdims=True)  # Shape (n2, 1)
    
    # Avoid division by zero
    mat1 = mat1 / (mat1_norms + 1e-8)
    mat2 = mat2 / (mat2_norms + 1e-8)
    
    # Compute cosine similarity
    similarity_matrix = np.dot(mat1, mat2.T)  # Shape (n1, n2)
    return similarity_matrix

def xml_to_json(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the root element's tag and attributes to confirm the file has been correctly loaded
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}

        # Use namespace
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": node.find("./data[@key='d0']", namespace).text.strip('"')
                if node.find("./data[@key='d0']", namespace) is not None
                else "",
                "description": node.find("./data[@key='d1']", namespace).text
                if node.find("./data[@key='d1']", namespace) is not None
                else "",
                "source_id": node.find("./data[@key='d2']", namespace).text
                if node.find("./data[@key='d2']", namespace) is not None
                else "",
            }
            data["nodes"].append(node_data)

        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": float(edge.find("./data[@key='d3']", namespace).text)
                if edge.find("./data[@key='d3']", namespace) is not None
                else 0.0,
                "description": edge.find("./data[@key='d4']", namespace).text
                if edge.find("./data[@key='d4']", namespace) is not None
                else "",
                "keywords": edge.find("./data[@key='d5']", namespace).text
                if edge.find("./data[@key='d5']", namespace) is not None
                else "",
                "source_id": edge.find("./data[@key='d6']", namespace).text
                if edge.find("./data[@key='d6']", namespace) is not None
                else "",
            }
            data["edges"].append(edge_data)

        # Print the number of nodes and edges found
        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None