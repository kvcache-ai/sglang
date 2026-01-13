#!/usr/bin/env python3
"""
Convert PEFT format LoRA adapter to SGLang format.

This script copies an entire LoRA adapter directory and converts the safetensors
weight files from PEFT format to SGLang format by removing unnecessary prefixes.
"""

import argparse
import shutil
from pathlib import Path
from safetensors.torch import load_file, save_file


def convert_lora_adapter(adapter_path: str, output_path: str = None, verbose: bool = True):
    """
    Convert PEFT format LoRA adapter to SGLang format.

    Args:
        adapter_path: Path to the input PEFT adapter directory
        output_path: Path to the output SGLang adapter directory (optional)
                     If None, defaults to {adapter_path}_converted
        verbose: Whether to print detailed conversion information
    """
    adapter_path = Path(adapter_path)

    # Set default output path if not provided
    if output_path is None:
        output_path = Path(str(adapter_path) + "_converted")
    else:
        output_path = Path(output_path)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    if not adapter_path.is_dir():
        raise ValueError(f"Adapter path must be a directory: {adapter_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 80)
        print("Converting PEFT LoRA Adapter to SGLang Format")
        print("=" * 80)
        print(f"\nInput:  {adapter_path}")
        print(f"Output: {output_path}\n")

    # Process all files in the adapter directory
    converted_files = []
    copied_files = []

    for file_path in adapter_path.iterdir():
        if file_path.is_file():
            output_file = output_path / file_path.name

            # Convert safetensors files
            if file_path.suffix == '.safetensors':
                if verbose:
                    print(f"Converting: {file_path.name}")

                # Load PEFT weights
                state_dict = load_file(str(file_path))

                # Convert keys by removing PEFT prefixes
                new_state_dict = {}
                for key, value in state_dict.items():
                    # Remove 'base_model.model.' prefix
                    new_key = key.replace("base_model.model.", "")
                    # Remove 'orig_module.' occurrences
                    new_key = new_key.replace(".orig_module", "")
                    # Remove '.default' adapter name (PEFT multi-adapter naming)
                    new_key = new_key.replace(".lora_A.default.", ".lora_A.")
                    new_key = new_key.replace(".lora_B.default.", ".lora_B.")
                    new_state_dict[new_key] = value

                    if verbose and key != new_key:
                        print(f"  {key}")
                        print(f"  -> {new_key}")

                # Save converted weights
                save_file(new_state_dict, str(output_file))
                converted_files.append(file_path.name)

                if verbose:
                    print(f"  Saved to: {output_file}\n")

            # Copy other files as-is
            else:
                if verbose:
                    print(f"Copying: {file_path.name}")
                shutil.copy2(str(file_path), str(output_file))
                copied_files.append(file_path.name)

    # Print summary
    if verbose:
        print("=" * 80)
        print("Conversion Summary")
        print("=" * 80)
        print(f"\nConverted files ({len(converted_files)}):")
        for f in converted_files:
            print(f"  - {f}")

        print(f"\nCopied files ({len(copied_files)}):")
        for f in copied_files:
            print(f"  - {f}")

        print(f"\nTotal: {len(converted_files) + len(copied_files)} files processed")
        print(f"Output directory: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PEFT format LoRA adapter to SGLang format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default output path (adds _converted suffix)
  python convert_lora.py /path/to/adapter

  # Specify custom output path
  python convert_lora.py /path/to/adapter /path/to/output

  # Quiet mode (minimal output)
  python convert_lora.py /path/to/adapter --quiet
        """
    )

    parser.add_argument(
        "adapter_path",
        type=str,
        help="Path to the input PEFT adapter directory"
    )

    parser.add_argument(
        "output_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the output SGLang adapter directory (default: {adapter_path}_converted)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - minimal output"
    )

    args = parser.parse_args()

    try:
        convert_lora_adapter(
            adapter_path=args.adapter_path,
            output_path=args.output_path,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
