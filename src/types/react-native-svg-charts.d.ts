declare module 'react-native-svg-charts' {
  import * as React from 'react';
  import { ViewStyle } from 'react-native';
  import { PathProps } from 'react-native-svg';

  export interface ChartProps {
    style?: ViewStyle;
    data: number[] | { value: number }[];
    svg?: Partial<PathProps>;
    contentInset?: { top?: number; bottom?: number; left?: number; right?: number };
    children?: React.ReactNode;
  }

  export interface XAxisProps extends ChartProps {
    formatLabel?: (value: any, index: number) => string; // ✅ fix
    numberOfTicks?: number;
    contentInset?: { left?: number; right?: number };
  }

  export interface YAxisProps extends ChartProps {
    formatLabel?: (value: any, index: number) => string;
    numberOfTicks?: number;
  }

  export class LineChart extends React.Component<ChartProps> {}
  export class Grid extends React.Component {}
  export class XAxis extends React.Component<XAxisProps> {}
  export class YAxis extends React.Component<YAxisProps> {}
}
