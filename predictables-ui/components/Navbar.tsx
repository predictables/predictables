import React from "react";
import Link from "next/link";
import Button from "@components/Button";
import navbarData from "@data/navbarData";
import NavItem from "@enums/NavItem";
import Dropdown from "./Dropdown";

interface Child {
  name: string;
  type: NavItem;
  route?: string;
  children?: Child[];
}

const Navbar = () => {
  const navClasses = `
  w-[100vw] h-fit flex flex-row p-5
  border-2 border-white
  fixed top-0 justify-center items-center`;

  return (
    <nav className={navClasses}>
      <div id="navbar-left" className="text-left w-[20%]">
        <ul className="justify-center">
          <Button>
            <Link href={"/"} className="p-3 justify-center">
              Home
            </Link>
          </Button>
        </ul>
      </div>
      <div id="navbar-right" className="text-right w-[80%]">
        <ul className="flex flex-row items-end justify-end">
          {Object.keys(navbarData).map((category, big_idx) => {
            let items = [];
            let childName;
            navbarData[category].map((child, idx) => {
              childName = child.name;
              child.children.map((child2, childidx2) => {
                items.push(child2);
              });
              return items;
            });
            return (
              <li key={big_idx}>
                <Dropdown title={childName} items={items} leftAlign={false} />
              </li>
            );
          })}
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
