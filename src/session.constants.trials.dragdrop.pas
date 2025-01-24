unit session.constants.trials.dragdrop;

{$mode ObjFPC}{$H+}

interface

uses session.constants.trials;

type
  TDragDropKeys = record
    ReleaseFoodForIntermediateHitsKey : string;
    AutoAnimateOnStartKey : string;
    DragDropOrientationKey : string;
    UseHelpProgressionKey : string;
    DistanceKey : string;
    DragModeKey : string;
    DragMoveFactorKey : string;
    DragableAnimationKey : string;
    GridSizeKey : string;
    NameKey : string;
    ReferenceNameKey : string;
    StimuliFolderKey : string;
  end;

const
  DragDropKeys : TDragDropKeys = (
    ReleaseFoodForIntermediateHitsKey : 'ReleaseFoodForIntermediateHits';
    AutoAnimateOnStartKey : 'AutoAnimateOnStart';
    DragDropOrientationKey : 'Orientation';
    UseHelpProgressionKey : 'UseHelpProgression';
    DistanceKey : 'Distance';
    DragModeKey : 'DragMode';
    DragMoveFactorKey : 'DragMoveFactor';
    DragableAnimationKey : 'DragableAnimation';
    GridSizeKey : 'GridSize';
    NameKey : HeaderName;
    ReferenceNameKey : HeaderReferenceName;
    StimuliFolderKey : 'StimuliFolder');

implementation

end.


