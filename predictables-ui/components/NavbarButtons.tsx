import Link from 'next/link';
import Button from '@components/Button';
import { data as dat } from '@data/navbarData';
import { NavButton } from '@data/navbarData';

interface NavbarButtonsProps {
  data: NavButton[];
  timeout?: number;
}

const NavbarButtons = ({ data = dat, timeout = 5000 }: NavbarButtonsProps) => {
  return (
    <>
      <ul className="flex flex-row items-end justify-end">
        {data.map((item: NavButton, idx: number) => {
          return (
            <li key={idx}>
              <Link href={item.route} passHref>
                <Button>{item.name}</Button>
              </Link>
            </li>
          );
        })}
      </ul>
    </>
  );
};

export default NavbarButtons;
