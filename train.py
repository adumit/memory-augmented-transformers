import torch

from memory_augmented_transformers.run_experiment import run_an_experiment


if __name__ == "__main__":
    rand_state = 1
    torch.manual_seed(rand_state)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # TODO: Ask for Dagshub token
    run_an_experiment(run_name="Test_run", device=device)
    