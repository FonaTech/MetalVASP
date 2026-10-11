/* Unique Mach-O entry point for Instruments. Avoid LaunchServices resolving
   the common Python bundle identifier/symlink to another installed Python.
   execv preserves this one launched process's PID and uses the exact argv. */
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: coreml-probe-launcher /absolute/python /absolute/probe.py [args]\n");
        return 2;
    }
    if (argv[1][0] != '/' || argv[2][0] != '/') {
        fprintf(stderr, "probe interpreter and script must be absolute paths\n");
        return 2;
    }
    printf("CoreML trace launcher pid=%ld interpreter=%s script=%s\n", (long)getpid(), argv[1], argv[2]);
    fflush(stdout);
    execv(argv[1], argv + 1);
    perror("execv exact CoreML probe interpreter");
    return 127;
}
