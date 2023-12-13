interface SidebarDataItem {
  name: string;
  href: string;
}

interface SidebarDataSection {
  name: string;
  children: SidebarDataItem[];
}

const sidebarData: SidebarDataSection[] = [
  {
    name: 'Load Data',
    children: [
      {
        name: 'Load from file',
        href: '/load/file',
      },
    ],
  },
];

export default sidebarData;
