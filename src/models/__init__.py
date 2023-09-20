from ._base import FlanT5
# the relevance is a singlevector
from .RelPromptQG import RelPromptFlanT5
# the relevance is a multivector
from .promptRelQG import SoftRelPromptFlanT5
# the encoder_output['last_hidden_states'] is a multivector with fixed-length
# the vector is prepresented by relevance and document and instruction
from .promptRelDocQG import SoftRelPromptDocFlanT5
