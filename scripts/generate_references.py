import subprocess
import os


def generate_references():
    try:
        # Run cffconvert and capture the output
        result = subprocess.run(
            ["cffconvert", "--format", "bibtex", "--infile", "CITATION.cff"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        bibtex_content = result.stdout

        # Modify the BibTeX entry
        # Replace the first line with the desired entry type and citation key
        bibtex_lines = bibtex_content.splitlines()
        if bibtex_lines:
            # Replace the first line
            bibtex_lines[0] = "@misc{Melton_AdaptiveResonanceLib_2024,"
            modified_bibtex = "\n".join(bibtex_lines)

            # Write the modified content to references.bib
            output_path = os.path.join("docs", "source", "artlib_citation.bib")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(modified_bibtex)
        else:
            print("Error: Empty BibTeX content.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running cffconvert: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise e


if __name__ == "__main__":
    generate_references()
