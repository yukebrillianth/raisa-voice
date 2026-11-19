import os

# The rules are tested on their own: an installed text tagger (compose mounts /models) must not
# change what these tests expect. test_text_tagger.py fakes one where it needs one.
os.environ["TEXT_TAGGER_DIR"] = "/nonexistent"
