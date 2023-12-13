import Button from "@components/Button";
import FileImport from "@components/FileImport";
import React from "react";

const LoadDataPage = () => {
  return (
    <div className="flex flex-row items-start h-full w-full justify-around border-2 border-black">
      <div className="flex flex-col justify-center items-center">
        <Button>Load Data</Button>
        <div className="my-5"></div>
        <FileImport />
      </div>
      <Button>Toy Dataset</Button>
    </div>
  );
};

export default LoadDataPage;
