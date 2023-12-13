import AccordionItem from './AccordionItem';
import AccordionChild from './AccordionChild';

interface DrawerSection {
  accordionTitle?: string;
  accordionType?: 'radio' | 'checkbox';
  isChecked?: boolean;
  accordionNumber?: number;
  children?: React.ReactNode;
  status?: boolean;
}

const DrawerSection = ({
  accordionTitle = 'Title goes here',
  accordionType = 'radio',
  isChecked = false,
  accordionNumber = 1,
  children,
  status = false,
}: DrawerSection) => {
  return (
    <div>
      <AccordionItem
        accordionTitle={accordionTitle}
        accordionType={accordionType}
        isChecked={isChecked}
        accordionNumber={accordionNumber}
        status={status}
      >
        <AccordionChild>{children}</AccordionChild>
      </AccordionItem>
    </div>
  );
};

export default DrawerSection;
