import Link from 'next/link';
import Button from '@components/Button';
import { data as dat } from '@data/navbarData';
import { NavButton } from '@data/navbarData';

interface NavbarButtonsProps {
  data: NavButton[];
  timeout?: number;
  isDataLoaded?: boolean;
}

const NavbarButtons = ({
  data = dat,
  timeout = 5000,
  isDataLoaded = false,
}: NavbarButtonsProps) => {
  return (
    <>
      <ul className="flex flex-row items-end justify-end">
        {data.map((item: NavButton, idx: number) => {
          return (
            <li key={idx}>
              <Link href={item.route} passHref>
                <Button inactive={item.name === 'view data' && !isDataLoaded}>
                  {item.name}
                </Button>
              </Link>
            </li>
          );
        })}
      </ul>
    </>
  );
};

export default NavbarButtons;
