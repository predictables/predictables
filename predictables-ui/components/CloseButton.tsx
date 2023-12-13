import { IoIosClose } from 'react-icons/io';

interface CloseButtonProps {
  onClick?: () => void;
}

const CloseButton = ({ onClick }: CloseButtonProps) => {
  return (
    <IoIosClose
      className={`h-[50px] w-[50px]
      bg-white text-black fill-black
      hover:text-white hover:bg-black hover:fill-white
      cursor-default active:scale-90
      
      text-center text-2xl justify-center items-center flex
    border-black border-[2px] shadow-lg
    
    `}
      onClick={onClick}
    />
  );
};

export default CloseButton;
