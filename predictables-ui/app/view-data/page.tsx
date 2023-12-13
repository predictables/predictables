import Table from '@plotting/Table';

interface ViewDataPageProps {
  data: any;
}

const ViewDataPage = ({ data }: ViewDataPageProps) => {
  let tbl = new Table(
    'view-data-page-table', // elementId
    [], // data
    [], // columnNames
    [], // rowNames
  );

  // Draw the table
  tbl.draw();

  return <div id="view-data-page"></div>;
};

export default ViewDataPage;
