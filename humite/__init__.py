"""OmniX: one autoregressive transformer for jets and collision events."""

from dotenv import find_dotenv, load_dotenv

# data and output directories may be set in a .env file of the working directory
load_dotenv(find_dotenv(usecwd=True))
