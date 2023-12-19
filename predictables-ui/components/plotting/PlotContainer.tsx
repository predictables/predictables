import { PlotContainerProps } from './interfaces';

const PlotContainer = ({
  height = 500,
  width = 750,
  children,
}: PlotContainerProps) => {
  return (
    <div
      className={`
        border-gray-200 border-[2px]
        rounded-xl shadow-lg
        
        duration-200
        hover:scale-[1.02]
        active:scale-[0.98]
      `}
    >
      <svg height={height} width={width}>
        {children}
      </svg>
    </div>
  );
};

export default PlotContainer;
