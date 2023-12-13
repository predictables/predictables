import FileImport from '@components/FileImport';
import { FaFileUpload } from 'react-icons/fa';
import Button from '@components/Button';
import { fetchPlaceholderData } from '@app/api/placeholder';
import Accordion from './Accordion';

interface LoadDataSectionProps {
  accordionTitle?: string;
  accordionType?: 'radio' | 'checkbox';
  isChecked?: boolean;
  setIsChecked?: any;
  accordionNumber?: number;
  status?: boolean;
  setStatus?: any;
  setLoadedData?: any;
}

const LoadDataSection = ({
  accordionType = 'checkbox',
  isChecked = false,
  setIsChecked,
  accordionNumber = 1,
  status = false,
  setStatus,
  setLoadedData,
}: LoadDataSectionProps) => {
  const accordionTitle = `load data`;
  const handleClick = async () => {
    try {
      // Use the mock function instead of the fetch API
      const data = await fetchPlaceholderData();

      // Assuming setLoadedData is a state setter function passed down as a prop
      if (data && setLoadedData) {
        setLoadedData(data);
        setStatus(true);
      }
    } catch (error) {
      console.error('Failed to load data:', error);
    }
  };

  return (
    <>
      <Accordion
        accordionTitle={accordionTitle}
        accordionType={accordionType}
        isChecked={isChecked}
        setIsChecked={setIsChecked}
        accordionNumber={accordionNumber}
        status={status}
        setStatus={setStatus}
      >
        <>
          <FileImport />
          <div className="my-3" />
          <Button onClick={handleClick}>
            <FaFileUpload
              className={`inline-block mr-2
            ${`bg-white text-black hover:bg-black hover:text-white shadow-lg scale-105`}
            `}
            />
            upload from file
          </Button>
        </>
      </Accordion>
    </>
  );
};

export default LoadDataSection;
