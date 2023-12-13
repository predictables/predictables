import LoadDataSection from './LoadDataSection';

interface DrawerChoicesProps {
  isExpanded?: boolean;
}

const DrawerChoices = ({ isExpanded = false }: DrawerChoicesProps) => {
  return (
    <>
      <li>
        <LoadDataSection isChecked={isExpanded} />
      </li>
    </>
  );
};

export default DrawerChoices;
