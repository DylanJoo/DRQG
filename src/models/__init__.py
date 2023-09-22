from ._base import FlanT5
from .RelPromptQG import RelPromptFlanT5

# the hidden state is prepresented by relevance and document and instruction
# the encoder_output['last_hidden_states'] is a multivector with fixed-length
from .promptRelDocQG import SoftRelPromptDocFlanT5
