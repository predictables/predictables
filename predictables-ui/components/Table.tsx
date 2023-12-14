import DataTable from '@models/DataTable/DataTable';

interface TableProps {
  dt: DataTable | null;
  maxRows?: number;
  maxCols?: number;
  renderIndex?: boolean;
}

/**
 * @component Table
 * @summary Renders a table of data from a DataTable object based on the original .render() method of a DataTable copied above.
 * @param {object} props - The component props.
 */
const Table = ({
  dt,
  maxRows = -1,
  maxCols = -1,
  renderIndex = true,
}: TableProps) => {
  // Returns an HTML table representation of the DataTable
  if (!dt) {
    return <div>dt is null</div>;
  } else {
    const { index, columns, values } = dt;

    const nRows = maxRows === -1 ? index.length : maxRows;
    const nCols = maxCols === -1 ? columns.length : maxCols;

    // Get the first maxRows rows of the index
    const renderedIndex = index.slice(0, nRows);

    // Get the first maxCols columns of the values
    const renderedCols = columns.slice(0, nCols);

    // Get the first maxRows rows x maxCols columns of the values
    const renderedValues = values.map((v: any) => v.slice(0, nRows, nCols));

    // Get the maximum length of the index and column names
    const maxIndexLength = Math.max(...renderedIndex.map((i: any) => i.length));
    const maxColLength = Math.max(...renderedCols.map((c: any) => c.length));

    // Pad the index and column names with spaces to make them all the same length
    renderedIndex.forEach((i: any, j: number) => {
      renderedIndex[j] = i.padEnd(maxIndexLength, ' ');
    });
    renderedCols.forEach((c: any, j: number) => {
      renderedCols[j] = c.padEnd(maxColLength, ' ');
    });

    return (
      <table className="h-full w-full">
        <thead>
          <tr>
            {renderIndex ? <th>{columns[0]}</th> : null}
            {renderedCols.map((col: any) => {
              <th>{col}</th>;
            })}
          </tr>
        </thead>
        <tbody>
          {renderedValues.map((row: any, i: number) => (
            <tr key={i}>
              {renderIndex ? <td>{renderedIndex[i]}</td> : null}
              {row.map((val: any, j: number) => (
                <td key={j}>{val}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
};

export default Table;
