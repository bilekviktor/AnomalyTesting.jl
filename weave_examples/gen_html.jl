M_DIR = normpath(@__DIR__)

#1st test
TEST1_PATH = normpath(M_DIR, "test1.jmd")
weave(TEST1_PATH, out_path = normpath(M_DIR, "test1"))
