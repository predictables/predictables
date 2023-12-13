import Logo from '@components/Logo';

interface DrawerButtonProps {
  onClick?: () => void;
}

const DrawerButton = ({ onClick }: DrawerButtonProps) => {
  return (
    <label htmlFor="drawer">
      <Logo hover={true} darkMode={false} inclIcon={true} onClick={onClick} />
    </label>
  );
};

export default DrawerButton;
