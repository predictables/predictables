'use client';

import { useState } from 'react';

interface AccordionProps {
  accordionTitle: string;
  accordionType?: 'radio' | 'checkbox';
  isChecked?: boolean;
  setIsChecked?: any;
  accordionNumber?: number;
  status?: boolean;
  setStatus?: any;
  children: React.ReactNode;
}

const Accordion = ({
  accordionTitle,
  accordionType = 'checkbox',
  isChecked = false,
  setIsChecked,
  accordionNumber = 1,
  status = false,
  setStatus,
  children,
}: AccordionProps) => {
  return (
    <>
      <div
        className={`collapse bg-base-200 cursor-pointer ${
          isChecked ? 'collapse-open' : 'collapse-closed'
        }`}
        onClick={() => setIsChecked(!isChecked)}
      >
        <input
          type={accordionType}
          name={accordionTitle.replace(' ', '-').toLowerCase()}
          checked={isChecked}
          readOnly
        />
        <div className="collapse-title text-xl font-medium">
          <h2 className="justify-around">
            {accordionTitle}
            <span className="text-base-content">
              {status ? ' - ✅' : ' - ❌'}
            </span>
          </h2>
        </div>
        <div className="collapse-content">{children}</div>
      </div>
    </>
  );
};

export default Accordion;
