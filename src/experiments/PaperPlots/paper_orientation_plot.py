import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from experiments.AsideExperiments.ex_orientation import experiment, plot_orientation
from experiments.VizReconstructionUtils import SpecialCellsPlotTuple
from experiments.subcell_paper.global_params import cred

data_manager = DataManager(
    path=config.paper_results_path,
    emissions_path=config.results_path,
    name='PaperOrientation',
    format=JOBLIB,
    trackCO2=True,
    country_alpha_code="FR"
)

lab = LabPipeline()

lab.define_new_block_of_functions(
    "experiment_orientation",
    experiment,
    recalculate=False
)
lab.execute(
    data_manager,
    num_cores=15,
    forget=False,
    save_on_iteration=None,
    num_cells_per_dim=[10, 30],  # 60
    image=[
        "batata.jpg",
    ],
    angle_threshold=[
        45
    ],
    method=["sobel"],
    kernel_size=[
        (3, 3),
    ]
)
plot_orientation(
    data_manager,
    path=config.subcell_paper_figures_path,
    method="sobel",
    num_cells_per_dim=10,
    image="batata.jpg",
    angle_threshold=45,
    alpha=0.5,
    format=".pdf",
    plot_by=["num_cells_per_dim"],
    numbers_on=True,
    specific_cells=[SpecialCellsPlotTuple(name="SpecialCell", indexes=[(7, 8)],
                                          color=cred, alpha=0.5)]
)

plot_orientation(
    data_manager,
    path=config.subcell_paper_figures_path,
    method="sobel",
    num_cells_per_dim=30,
    image="batata.jpg",
    angle_threshold=45,
    alpha=0.5,
    format=".pdf",
    plot_by=["num_cells_per_dim"],
    numbers_on=True,
)
