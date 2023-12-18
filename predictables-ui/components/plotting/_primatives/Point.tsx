interface pointProps {
  x: number;
  y: number;
  radius: number;
  edgeColor: string;
  fillColor: string;
}

const Point = ({ x, y, radius, edgeColor, fillColor }: pointProps) => {
  return (
    <circle
      cx={x}
      cy={y}
      r={radius}
      stroke={edgeColor}
      fill={fillColor}
    ></circle>
  );
};

export default Point;
