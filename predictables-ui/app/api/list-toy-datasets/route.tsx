import fs from "fs";
import path from "path";
import { NextRequest, NextResponse } from "next/server";

const getToyDataFiles = () => {
  const toyDataFolder = "@data/";
  try {
    return fs
      .readdirSync(toyDataFolder)
      .map((file) => path.join(toyDataFolder, file));
  } catch (err) {
    console.log("There was an error reading the toy data folder: ", err);
    return [];
  }
};

const fmtToyDataFiles = (files: string[]) => {
  let files1 = files.map((file: string) => file.split("/").slice(-1)[0]);
  files1 = files1.map((file: string) => file.split(".").slice(0, -1)[0]);
  return files1;
};

export const GET = async (request: NextRequest) => {
  if (request.method === "GET") {
    const files0: string[] = getToyDataFiles();
    const files: string[] = fmtToyDataFiles(files0);

    return new NextResponse(JSON.stringify(files), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } else {
    return new NextResponse(
      JSON.stringify({
        message: "Method not allowed",
      }),
      {
        status: 405,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
};
