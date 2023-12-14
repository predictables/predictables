import DataTable from '@models/DataTable/DataTable';

// Create an interface for the context value
export interface DataTableContextType {
  dt: DataTable | null;
  setDT: React.Dispatch<React.SetStateAction<DataTable | null>>;
}
