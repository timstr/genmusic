import sys


def progress_bar(current, total, message):
    if not sys.stdout.isatty():
        return
    i = current + 1
    bar_fill = "=" * (i * 50 // total)
    sys.stdout.write("\r[%-50s] %d/%d" % (bar_fill, i, total))
    if (message is not None) and (len(message) > 0):
        sys.stdout.write(" ")
        sys.stdout.write(message)
    if i == total:
        sys.stdout.write("\n")
    sys.stdout.flush()
