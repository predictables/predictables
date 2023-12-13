'use client';

import { useState } from 'react';

interface AccordionItemProps {
  accordionTitle: string;
  accordionType?: 'radio' | 'checkbox';
  isChecked?: boolean;
  accordionNumber?: number;
  children?: React.ReactNode;
}

const AccordionItem = ({
  accordionTitle,
  accordionType = 'radio',
  isChecked = false,
  accordionNumber = 1,
  children,
}: AccordionItemProps) => {
  const [isExpanded, setIsExpanded] = useState(isChecked);

  const handleClick = () => {
    const curState = isExpanded;
    setIsExpanded(!curState);
  };

  return (
    <div
      className={`
      collapse collapse-arrow bg-base-200
      ${isExpanded ? 'collapse-open' : 'collapse-closed'} 
      `}
      onClick={handleClick}
    >
      <input
        type={accordionType}
        name={accordionTitle.toLowerCase().replace(' ', '-')}
        checked={isChecked}
        data-accordion={accordionNumber}
      />
      <h2 className="collapse-title text-xl font-medium">{accordionTitle}</h2>
      <div className="collapse-content">{children}</div>
    </div>
  );
};

export default AccordionItem;
