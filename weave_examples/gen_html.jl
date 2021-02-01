using Weave
M_DIR = normpath(@__DIR__)

#1st test
TEST1_PATH = normpath(M_DIR, "test1.jmd")
weave(TEST1_PATH, out_path = normpath(M_DIR, "test1"))

#2nd test
TEST2_PATH = normpath(M_DIR, "test2.jmd")
weave(TEST2_PATH, out_path = normpath(M_DIR, "test2"))

#3rd test
TEST3_PATH = normpath(M_DIR, "test3.jmd")
weave(TEST3_PATH, out_path = normpath(M_DIR, "test3"))

TEST3_8D_PATH = normpath(M_DIR, "test3_8D.jmd")
weave(TEST3_8D_PATH, out_path = normpath(M_DIR, "test3_8D"))

TEST4_1_PATH = normpath(M_DIR, "test4.1.jmd")
weave(TEST4_1_PATH, out_path = normpath(M_DIR, "test4"))

TEST4_2_PATH = normpath(M_DIR, "test4.2.jmd")
weave(TEST4_2_PATH, out_path = normpath(M_DIR, "test4"))

TEST5_1_PATH = normpath(M_DIR, "test5.1.jmd")
weave(TEST5_1_PATH, out_path = normpath(M_DIR, "test5"))

TEST5_2_PATH = normpath(M_DIR, "test5.2.jmd")
weave(TEST5_2_PATH, out_path = normpath(M_DIR, "test5"))

TEST5_3_PATH = normpath(M_DIR, "test5.3.jmd")
weave(TEST5_3_PATH, out_path = normpath(M_DIR, "test5"))

TEST6_1_PATH = normpath(M_DIR, "test6.1.jmd")
weave(TEST6_1_PATH, out_path = normpath(M_DIR, "test6"))

TEST6_2_PATH = normpath(M_DIR, "test6.2.jmd")
weave(TEST6_2_PATH, out_path = normpath(M_DIR, "test6"))
