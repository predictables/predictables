import React from "react";
import ButtonTypes from "@enums/ButtonTypes";

// interface FileImportProps {
//   buttonStyle?: ButtonTypes;
// }

const FileImport = () => {
  return (
    <input
      type="file"
      className="file-input file-input-bordered w-full max-w-xs "
    />
  );
};

export default FileImport;
