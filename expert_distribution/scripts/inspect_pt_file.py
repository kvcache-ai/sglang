#!/usr/bin/env python3
"""Interactive .pt file inspector.

Load a .pt file and interactively inspect tensor contents with slicing support.

Usage:
    python inspect_pt_file.py <pt_file_path>
"""

import argparse
import sys
from typing import Any, Dict, List, Optional, Union

import torch


def print_structure(data: Any, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
    """Recursively print the structure of loaded data."""
    indent = "  " * current_depth

    if isinstance(data, torch.Tensor):
        print(f"{indent}{prefix}Tensor: shape={list(data.shape)}, dtype={data.dtype}, device={data.device}")
    elif isinstance(data, dict):
        print(f"{indent}{prefix}Dict with {len(data)} keys:")
        if current_depth < max_depth:
            for key, value in data.items():
                print_structure(value, prefix=f"['{key}']: ", max_depth=max_depth, current_depth=current_depth + 1)
    elif isinstance(data, (list, tuple)):
        type_name = "List" if isinstance(data, list) else "Tuple"
        print(f"{indent}{prefix}{type_name} with {len(data)} items")
        if current_depth < max_depth and len(data) > 0:
            # Show first item structure
            print_structure(data[0], prefix="[0]: ", max_depth=max_depth, current_depth=current_depth + 1)
            if len(data) > 1:
                print(f"{indent}  ... ({len(data) - 1} more items)")
    else:
        type_name = type(data).__name__
        if isinstance(data, (int, float, str, bool, type(None))):
            print(f"{indent}{prefix}{type_name}: {data}")
        else:
            print(f"{indent}{prefix}{type_name}")


def get_nested_value(data: Any, path: str) -> Any:
    """Get a nested value from data using a path string.

    Path examples:
        "key1" -> data["key1"]
        "key1.key2" -> data["key1"]["key2"]
        "key1[0]" -> data["key1"][0]
        "key1[0].key2" -> data["key1"][0]["key2"]
    """
    if not path:
        return data

    current = data
    # Parse path: split by '.' but handle [...] brackets
    tokens = []
    i = 0
    current_token = ""

    while i < len(path):
        char = path[i]
        if char == '.':
            if current_token:
                tokens.append(current_token)
                current_token = ""
        elif char == '[':
            if current_token:
                tokens.append(current_token)
                current_token = ""
            # Find matching ]
            j = i + 1
            while j < len(path) and path[j] != ']':
                j += 1
            bracket_content = path[i+1:j]
            tokens.append(f"[{bracket_content}]")
            i = j
        else:
            current_token += char
        i += 1

    if current_token:
        tokens.append(current_token)

    # Navigate through tokens
    for token in tokens:
        if token.startswith('[') and token.endswith(']'):
            # Index access
            index_str = token[1:-1]
            try:
                index = int(index_str)
                current = current[index]
            except ValueError:
                # String key in brackets
                key = index_str.strip("'\"")
                current = current[key]
        else:
            # Dict key access
            current = current[token]

    return current


def parse_slice(slice_str: str) -> Union[int, slice, tuple]:
    """Parse a slice string into a slice object or tuple of slices.

    Examples:
        "0" -> 0
        ":" -> slice(None)
        "0:10" -> slice(0, 10)
        "0, :" -> (0, slice(None))
        "0:5, 1:3, :" -> (slice(0, 5), slice(1, 3), slice(None))
    """
    # Split by comma for multi-dimensional slicing
    parts = [p.strip() for p in slice_str.split(',')]

    def parse_single(s: str):
        s = s.strip()
        if ':' not in s:
            # Single index
            return int(s)
        else:
            # Slice
            components = s.split(':')
            if len(components) == 2:
                start = int(components[0]) if components[0] else None
                stop = int(components[1]) if components[1] else None
                return slice(start, stop)
            elif len(components) == 3:
                start = int(components[0]) if components[0] else None
                stop = int(components[1]) if components[1] else None
                step = int(components[2]) if components[2] else None
                return slice(start, stop, step)
            else:
                raise ValueError(f"Invalid slice: {s}")

    if len(parts) == 1:
        return parse_single(parts[0])
    else:
        return tuple(parse_single(p) for p in parts)


def interactive_inspect(data: Any, data_path: str = "data"):
    """Interactive inspection loop."""
    print("\n" + "=" * 60)
    print("Interactive Inspector")
    print("=" * 60)
    print("Commands:")
    print("  path           - Show value at path (e.g., 'gpu_hit_ratio' or 'key1.key2[0]')")
    print("  path[slice]    - Show tensor slice (e.g., 'tensor[0:5, :]' or 'key[0, 1:10]')")
    print("  .stats path    - Show tensor statistics")
    print("  .shape path    - Show tensor shape")
    print("  .struct        - Show data structure again")
    print("  .keys          - Show top-level keys (if dict)")
    print("  .help          - Show this help")
    print("  .quit / .exit  - Exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        # Handle special commands
        if user_input in ('.quit', '.exit', 'quit', 'exit'):
            print("Exiting...")
            break
        elif user_input == '.help':
            print("Commands:")
            print("  path           - Show value at path")
            print("  path[slice]    - Show tensor slice")
            print("  .stats path    - Show tensor statistics")
            print("  .shape path    - Show tensor shape")
            print("  .struct        - Show data structure")
            print("  .keys          - Show top-level keys")
            print("  .quit          - Exit")
            continue
        elif user_input == '.struct':
            print("\nData structure:")
            print_structure(data)
            continue
        elif user_input == '.keys':
            if isinstance(data, dict):
                print(f"Keys: {list(data.keys())}")
            else:
                print(f"Data is not a dict, it's a {type(data).__name__}")
            continue
        elif user_input.startswith('.stats '):
            path = user_input[7:].strip()
            try:
                value = get_nested_value(data, path)
                if isinstance(value, torch.Tensor):
                    print(f"Statistics for '{path}':")
                    print(f"  Shape: {list(value.shape)}")
                    print(f"  Dtype: {value.dtype}")
                    print(f"  Min: {value.min().item()}")
                    print(f"  Max: {value.max().item()}")
                    print(f"  Mean: {value.float().mean().item():.6f}")
                    print(f"  Std: {value.float().std().item():.6f}")
                    print(f"  Sum: {value.sum().item()}")
                    # Count non-zero
                    nonzero = (value != 0).sum().item()
                    print(f"  Non-zero count: {nonzero} / {value.numel()} ({100*nonzero/value.numel():.2f}%)")
                else:
                    print(f"'{path}' is not a tensor")
            except Exception as e:
                print(f"Error: {e}")
            continue
        elif user_input.startswith('.shape '):
            path = user_input[7:].strip()
            try:
                value = get_nested_value(data, path)
                if isinstance(value, torch.Tensor):
                    print(f"Shape of '{path}': {list(value.shape)}")
                else:
                    print(f"'{path}' is not a tensor")
            except Exception as e:
                print(f"Error: {e}")
            continue

        # Parse path and optional slice
        try:
            # Check if there's a slice at the end
            if '[' in user_input:
                # Find the last '[' that could be a slice
                # Need to distinguish between path brackets and slice brackets
                # Heuristic: if it ends with ']', check if it's a slice pattern
                last_bracket = user_input.rfind('[')
                potential_slice = user_input[last_bracket+1:-1] if user_input.endswith(']') else None

                # Check if potential_slice looks like a slice (contains ':' or ',')
                if potential_slice and (':' in potential_slice or ',' in potential_slice):
                    path = user_input[:last_bracket]
                    slice_str = potential_slice
                else:
                    # It's part of the path
                    path = user_input
                    slice_str = None
            else:
                path = user_input
                slice_str = None

            # Get the value
            value = get_nested_value(data, path)

            if slice_str:
                # Apply slice
                if isinstance(value, torch.Tensor):
                    slice_obj = parse_slice(slice_str)
                    sliced = value[slice_obj]
                    print(f"Shape after slicing: {list(sliced.shape) if isinstance(sliced, torch.Tensor) else 'scalar'}")
                    print(sliced)
                else:
                    print(f"Cannot slice non-tensor: {type(value).__name__}")
            else:
                # Just print the value
                if isinstance(value, torch.Tensor):
                    print(f"Tensor: shape={list(value.shape)}, dtype={value.dtype}")
                    if value.numel() <= 100:
                        print(value)
                    else:
                        print(f"(Tensor too large to display, use slicing. Total elements: {value.numel()})")
                elif isinstance(value, dict):
                    print(f"Dict with keys: {list(value.keys())}")
                elif isinstance(value, (list, tuple)):
                    print(f"{'List' if isinstance(value, list) else 'Tuple'} with {len(value)} items")
                    if len(value) <= 10:
                        for i, item in enumerate(value):
                            if isinstance(item, torch.Tensor):
                                print(f"  [{i}]: Tensor shape={list(item.shape)}")
                            else:
                                print(f"  [{i}]: {type(item).__name__}")
                else:
                    print(value)

        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive .pt file inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python inspect_pt_file.py model.pt

Interactive commands:
    gpu_hit_ratio           - Access dict key 'gpu_hit_ratio'
    gpu_hit_ratio[0:5, :]   - Slice tensor
    .stats gpu_hit_ratio    - Show tensor statistics
    .shape gpu_hit_ratio    - Show tensor shape
    .struct                 - Show data structure
    .keys                   - Show top-level keys
    .quit                   - Exit
        """
    )
    parser.add_argument("--pt_file", type=str, help="Path to .pt file", default="/tmp/expert_distribution_recorder_1767180841.5184014.pt")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Only print structure, don't enter interactive mode",
    )

    args = parser.parse_args()

    # Load file
    print(f"Loading: {args.pt_file}")
    try:
        data = torch.load(args.pt_file, weights_only=False, map_location="cpu")
    except Exception as e:
        print(f"Error loading file: {e}")
        return 1

    # Print structure
    print("\nData structure:")
    print_structure(data)

    # Interactive mode
    if not args.no_interactive:
        interactive_inspect(data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
