import os
import gzip
import shutil
import csv
import re
import pandas as pd


def get_data(file_name: str) -> pd.DataFrame:
    with open(file_name, "r") as f:
        raw_data = pd.read_csv(f, sep="\t", dtype={"IDSubtitleFile": str, "SubHearingImpaired": bool})
        print(type(raw_data["IDSubtitleFile"]))
        return raw_data


def get_path(export_data: pd.DataFrame) -> list[list[str]]:
    path = []
    local = "C:/Users/binz3/Dokumente/Uni/STeM/Semester 3/Methoden der Datenanalyse/Hausarbeit/Subtitles/export_custom/files"
    i = 1
    for sub_id in export_data["IDSubtitleFile"]:
        print(sub_id)
        print(i)
        i += 1
        p = ""
        try:
            if len(sub_id) >= 4:
                p += "/" + sub_id[-1] + "/" + sub_id[-2] + "/" + sub_id[-3] + "/" + sub_id[-4]
            elif len(sub_id) == 3:
                p += "/" + sub_id[-1] + "/" + sub_id[-2] + "/" + sub_id[-3]
            elif len(sub_id) == 2:
                p += "/" + sub_id[-1] + "/" + sub_id[-2]
            else:
                p += "/" + sub_id
        except TypeError:
            continue
        p = [local + p, sub_id]
        path.append(p)
    return path


def clean_srt(path, file):
    new_lines = []
    with open(path + file + ".srt", "r", encoding="ansi") as f:
        lines = f.readlines()
        for line in lines:
            if not line[:-1].isnumeric():
                if not re.match(r"\d\d:\d\d:\d\d[,\.]\d\d\d", line):
                    if not re.match(r"[\(\[]\w*[\)\]]", line):
                        line = re.sub(r"^- ", "", line)
                        new_lines.append(line)
    with open(path + "/s_" + file + ".txt", "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    os.remove(path + file + ".srt")


def get_sub(paths: list[list[str]]) -> None:
    try:
        os.mkdir("./corpus/")
    except FileExistsError:
        pass
    i = len(paths)
    j = 0
    for path in paths:
        with gzip.open(path[0] + "/" + path[1] + ".gz", 'rb') as file_in:
            with open("./corpus/" + path[1] + ".srt", 'wb') as file_out:
                shutil.copyfileobj(file_in, file_out)
        # clean_srt("./corpus/", path[1])
        j += 1
        print("Successfully extracted " + str(j) + "/" + str(i))

    print("Extracted: " + str(j) + "/" + str(i))


def main():
    export_data = get_data("export-txt.csv")
    print(export_data)
    # del l[0]
    paths = get_path(export_data)
    # print(paths)
    get_sub(paths)


main()
