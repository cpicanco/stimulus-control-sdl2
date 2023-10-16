unit sdl.app.grids.types;

{$mode ObjFPC}{$H+}

interface

uses SDL2, fgl, Math.LatinSquares;

type
  TGridStyle =
     (gtCircle, gtSquare, gtDistributed);

   TGridOrientation =
     (goNone, goLeftToRight, goRightToLeft, goTopToBottom, goBottomToTop);

   TCell = array [0..1] of Integer;

   TGridList = specialize TFPGList<Integer>;
   TGridItem = record
     Index : integer;
     Position : integer;
     Rect : TSDL_Rect;
     Item : TObject;
   end;

   TGridItems = array of TGridItem;

   TMatrix = array of array of TGridItem;

   { TRandomPositions }

   TRandomPositions = record
     Samples: TGridItems;
     SamplesRows : TLatinSquare;
     Comparisons : TGridItems;
     ComparisonsRows : TLatinSquare;
   end;

implementation

end.
