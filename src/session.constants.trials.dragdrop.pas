{
  Stimulus Control
  Copyright (C) 2025-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
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


