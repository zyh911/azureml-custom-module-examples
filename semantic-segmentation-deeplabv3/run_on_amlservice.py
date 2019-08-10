import os

from tools.amlservice_scaffold.amlservice_pipeline import Module, PipelineStep, run_pipeline
from azureml.core import RunConfiguration, Workspace
from azureml.core.environment import DEFAULT_GPU_IMAGE


MODULE_SPECS_FOLDER = 'module_specs'


def spec_file_path(spec_file_name):
    return os.path.join(MODULE_SPECS_FOLDER, spec_file_name)


def get_workspace(name, subscription_id, resource_group):
    return Workspace.get(
        name=name,
        subscription_id=subscription_id,
        resource_group=resource_group
    )


def get_run_config(comp, compute_name, use_gpu=False):
    if comp.image:
        run_config = RunConfiguration()
        run_config.environment.docker.base_image = comp.image
    else:
        run_config = RunConfiguration(conda_dependencies=comp.conda_dependencies)
    run_config.target = compute_name
    run_config.environment.docker.enabled = True
    if use_gpu:
        run_config.environment.docker.base_image = DEFAULT_GPU_IMAGE
        run_config.environment.docker.gpu_support = True

    return run_config


def create_pipeline_steps(compute_name):
    # Load module spec from yaml file
    train = Module(
        spec_file_path=spec_file_path('train.yaml'),
        source_directory='script'
    )
    score = Module(
        spec_file_path=spec_file_path('score.yaml'),
        source_directory='script',
    )

    # Run config setting
    run_config_train = get_run_config(train, compute_name, use_gpu=True)
    run_config_score = get_run_config(score, compute_name, use_gpu=True)

    # Assign parameters
    train.params['Data path'].assign('dataset')
    train.params['Model depth'].assign(100)
    train.params['Growth rate'].assign(12)
    train.params['Memory efficient'].assign(False)
    train.params['Valid size'].assign(5000)
    train.params['Epochs'].assign(3)
    train.params['Batch size'].assign(64)
    train.params['Random seed'].assign(1)

    score.params['Data path'].assign('test_data')

    # Connect ports
    score.inputs['Model path'].connect(train.outputs['Save path'])

    # Convert to a list of PipelineStep, which can be ran by AML Service
    pipeline_step_list = [
        PipelineStep(train, run_config=run_config_train),
        PipelineStep(score, run_config=run_config_score)
    ]

    return pipeline_step_list


if __name__ == '__main__':
    workspace = get_workspace(
        name='yuhazh2',
        subscription_id='e9b2ec51-5c94-4fa8-809a-dc1e695e4896',
        resource_group='yuhazh'
    )
    compute_name = 'yuhazh-compute'
    pipeline_steps = create_pipeline_steps(compute_name)
    run_pipeline(steps=pipeline_steps, experiment_name='test-yaml-1', workspace=workspace)
