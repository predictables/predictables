import Logo from '@components/Logo';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <Logo inclTitle={true} hover={false} />
    </main>
  );
}
