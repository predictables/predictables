'use client';

import Table from '@components/Table';
import DataTable from '@models/DataTable/DataTable';

interface ViewDataPageProps {
  df: DataTable | null;
}

{
  /* <div className="h-[100vh] w-[100vw]"> */
}

const ViewDataPage = ({ df }: ViewDataPageProps) => {
  return <>{df ? <Table dt={df} /> : <div>df is null</div>}</>;
};

export default ViewDataPage;
