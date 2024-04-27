from pathlib import Path

project_root = Path(__file__).parent.parent

data_path = Path.joinpath(project_root, 'Data')
results_path = Path.joinpath(project_root, 'Results')
# paper_results_path = Path.joinpath(results_path, 'Paper')
paper_results_path = Path.joinpath(results_path, 'PaperTest')
data_path.mkdir(parents=True, exist_ok=True)
results_path.mkdir(parents=True, exist_ok=True)
paper_results_path.mkdir(parents=True, exist_ok=True)

subcell_paper_folder_path = Path.joinpath(project_root, 'paper')
subcell_paper_figures_path = Path.joinpath(subcell_paper_folder_path, 'figures')
subcell_presentation_path = Path.joinpath(results_path, 'Presentation')
subcell_paper_folder_path.mkdir(parents=True, exist_ok=True)
subcell_paper_figures_path.mkdir(parents=True, exist_ok=True)
subcell_presentation_path.mkdir(parents=True, exist_ok=True)

images_path = Path.joinpath(data_path, 'Images')
images_path.mkdir(parents=True, exist_ok=True)
