from study1_constants import participants_to_ignore as participants_to_ignore1
from study2_constants import participants_to_ignore as participants_to_ignore2
from study3_constants import participants_to_ignore as participants_to_ignore3
from study4_constants import participants_to_ignore as participants_to_ignore4
participants_to_ignore = participants_to_ignore1 + participants_to_ignore2 + participants_to_ignore3 + participants_to_ignore4

from study1_constants import participants_natural as participants_natural1
from study1_constants import participants_social as participants_social1
from study2_constants import participants_natural as participants_natural2
from study2_constants import participants_humanities as participants_social2
from study3_constants import participants as participants_natural3


participants_natural = participants_natural1 + participants_natural2 + participants_natural3
participants_social = participants_social1 + participants_social2

if __name__ == "__main__":
    print(len(participants_natural + participants_social))
    print(len(participants_natural))
    print(len(participants_social))