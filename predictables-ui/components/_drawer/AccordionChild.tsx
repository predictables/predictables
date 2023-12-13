interface AccordionChildProps {
  children: React.ReactNode;
}

const AccordionChild = ({ children }: AccordionChildProps) => {
  return <div>{children}</div>;
};

export default AccordionChild;
