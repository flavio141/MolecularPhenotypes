# MolecularPhenotypes
## Get Started
Let's start with a few point:
1. Run the `git clone` command followed by the URL of the main repository::

   git clone <link_to_repo> 

2. Now that you have cloned the main repository, navigate into the cloned directory using the `cd` command:

   cd MolecularPhenotypes

3. At this point, the main repository will not have the submodule's content yet. To also fetch the submodule's code, run the command:

   git submodule update --init --recursive

This command initializes and updates the submodule within the cloned repository.
You have cloned the main repository with its submodule. You can proceed with any necessary operations or modifications in your local environment.

Now, it is necessary to create a virtualEnv and install all the dependencies from the 'requirements.txt' file. Pay attention that pytorch is also listed inside the file, but it is better to install it before from the official website: https://pytorch.org/get-started/locally/

## Structure
Inside the folder: 'src' there is the main code. The 'preprocessing.py' file is useful in order to download all fasta and PDB information for the necessary protein.