{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.grids.types;

{$mode ObjFPC}{$H+}

interface

uses SDL2, Generics.Collections, Math.LatinSquares;

type
  TGridStyle =
     (gtCircle, gtSquare, gtDistributed);

   TGridOrientation = (goNone,
     goLeftToRight, goRightToLeft, goTopToBottom, goBottomToTop,
     goCustom);

   TCell = array [0..1] of Integer;

   TGridList = specialize TList<Integer>;
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

   TBorder = record
     Top : TSDL_Rect;
     Bottom : TSDL_Rect;
     Left : TSDL_Rect;
     Right: TSDL_Rect;
   end;

implementation

end.
