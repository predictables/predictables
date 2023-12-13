'use client';

import { useState } from 'react';

interface AccordionItemInputProps {
  title: string;
  type: 'radio' | 'checkbox';
  isChecked?: boolean;
  number?: number;
  status?: boolean;
}

const AccordionItemInput = ({
  title,
  type,
  isChecked = false,
  number = 1,
  status = false,
}: AccordionItemInputProps) => {
  const adjTitle = title.toLowerCase().replace(' ', '-');
  // const titlePlusUTF = `${adjTitle} ${status ? '✅' : '❌'}`;

  return (
    <input
      type={type}
      name={adjTitle}
      checked={isChecked}
      data-accordion={number}
      className="accordion"
      readOnly
    />
  );
};

interface AccordionItemProps {
  accordionTitle: string;
  accordionType?: 'radio' | 'checkbox';
  isChecked?: boolean;
  accordionNumber?: number;
  children?: React.ReactNode;
  status?: boolean;
}

const AccordionItem = ({
  accordionTitle,
  accordionType = 'radio',
  isChecked = false,
  accordionNumber = 1,
  children,
  status = false,
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
      <AccordionItemInput
        title={accordionTitle}
        type={accordionType}
        isChecked={isChecked}
        number={accordionNumber}
        status={status}
      />
      <h2 className="collapse-title text-xl font-medium">{accordionTitle}</h2>
      <div className="collapse-content">{children}</div>
    </div>
  );
};

export default AccordionItem;
