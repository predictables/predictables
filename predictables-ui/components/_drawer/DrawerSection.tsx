import AccordionItem from './AccordionItem';
import AccordionChild from './AccordionChild';

interface DrawerSection {
  accordionTitle?: string;
  accordionType?: 'radio' | 'checkbox';
  isChecked?: boolean;
  accordionNumber?: number;
  children?: React.ReactNode;
}

const DrawerSection = ({
  accordionTitle = 'Title goes here',
  accordionType = 'radio',
  isChecked = false,
  accordionNumber = 1,
  children,
}: DrawerSection) => {
  return (
    <div>
      <AccordionItem
        accordionTitle={accordionTitle}
        accordionType={accordionType}
        isChecked={isChecked}
        accordionNumber={accordionNumber}
      >
        <AccordionChild>{children}</AccordionChild>
      </AccordionItem>
    </div>
  );
};

export default DrawerSection;
