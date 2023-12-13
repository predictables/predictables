import FileImport from '@components/FileImport';
import DrawerSection from './DrawerSection';
import Link from 'next/link';
import { FaFileUpload } from 'react-icons/fa';
import Button from '@components/Button';

interface LoadDataSection {
  accordionTitle?: string;
  accordionType?: 'radio' | 'checkbox';
  isChecked?: boolean;
  accordionNumber?: number;
}

const LoadDataSection = ({
  accordionTitle = 'load data',
  accordionType = 'checkbox',
  isChecked = false,
  accordionNumber = 1,
}: DrawerSection) => {
  return (
    <>
      <DrawerSection
        accordionTitle={accordionTitle}
        accordionType={accordionType}
        isChecked={isChecked}
        accordionNumber={accordionNumber}
      >
        <FileImport />
        <Link href="/api/load/file">
          <Button>
            <FaFileUpload
              className={`inline-block mr-2
            ${`bg-white text-black hover:bg-black hover:text-white shadow-lg scale-105`}
            `}
            />
            upload from file
          </Button>
        </Link>
      </DrawerSection>
    </>
  );
};

export default LoadDataSection;
