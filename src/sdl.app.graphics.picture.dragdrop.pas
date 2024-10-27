{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.graphics.picture.dragdrop;

{$mode ObjFPC}{$H+}
{$modeswitch TypeHelpers}

interface

uses
  Classes, SysUtils
  , sdl.app.choiceable.picture
  , sdl.app.choiceable.rectangule
  , sdl.app.dragdropable.contract
  , sdl.app.events.abstract
  , sdl.app.stimulus.contract
  , sdl.app.stimulus.types
  , sdl.app.grids.types
  , SDL2
  , Math
  , math.bresenhamline.classes
  ;

type

  { TDragDropablePicture }

  TDragDropablePicture = class(TChoiceablePicture, IDragDropable, IStimulus)
  private
    FBresenhamLine : TBresenhamLine;
    FBorder : TBorder;
    FIsSample: Boolean;
    FPosition : integer;
    FDraggable: Boolean;
    FIsDragging : Boolean;
    FOnOtherDragDrop: TDragDropEvent;
    FOnRightDragDrop: TDragDropEvent;
    FOnWrongDragDrop: TDragDropEvent;
    FOffSet : TPoint;
    procedure SetDraggable(AValue: Boolean);
    procedure SetOnOtherDragDrop(AValue: TDragDropEvent);
    procedure SetOnRightDragDrop(AValue: TDragDropEvent);
    procedure SetOnWrongDragDrop(AValue: TDragDropEvent);
    procedure BorderCollision;
  protected
    procedure DragDrop(Sender, Source: TObject; X, Y: Integer);
    procedure MouseMove(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer); override;
    procedure MouseDown(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer); override;
    procedure MouseUp(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer); override;
    procedure MouseEnter(Sender: TObject); override;
    procedure MouseExit(Sender: TObject); override;
    procedure Paint; override;
  public
    constructor Create; override;
    destructor Destroy; override;
    procedure Load(AParameters : TStringList;
      AParent : TObject; ARect: TSDL_Rect);
    procedure Start;
    procedure Stop;
    procedure DoResponse(AHuman : Boolean);
    procedure UpdateDistance;
    procedure MoveToPoint(APorcentage: Float);
    function GetID: TStimulusID;
    function ToData : string;
    function ToJSON : string;
    property OnRightDragDrop : TDragDropEvent read FOnRightDragDrop write SetOnRightDragDrop;
    property OnWrongDragDrop : TDragDropEvent read FOnWrongDragDrop write SetOnWrongDragDrop;
    property OnOtherDragDrop : TDragDropEvent read FOnOtherDragDrop write SetOnOtherDragDrop;
    property Draggable : Boolean read FDraggable write SetDraggable;
    property Position : integer read FPosition write FPosition;
    property IsSample : Boolean read FIsSample write FIsSample;
  end;

implementation

uses
  session.pool,
  session.loggers.writerow.timestamp,
  sdl.app.grids.methods,
  sdl.helpers;

var SomeInstanceIsDragging : Boolean;

{ TChoiceablePicture }

procedure TDragDropablePicture.SetOnOtherDragDrop(AValue: TDragDropEvent);
begin
  if FOnOtherDragDrop=AValue then Exit;
  FOnOtherDragDrop:=AValue;
end;

procedure TDragDropablePicture.SetDraggable(AValue: Boolean);
begin
  if FDraggable=AValue then Exit;
  FDraggable:=AValue;
end;

procedure TDragDropablePicture.SetOnRightDragDrop(AValue: TDragDropEvent);
begin
  if FOnRightDragDrop=AValue then Exit;
  FOnRightDragDrop:=AValue;
end;

procedure TDragDropablePicture.SetOnWrongDragDrop(AValue: TDragDropEvent);
begin
  if FOnWrongDragDrop=AValue then Exit;
  FOnWrongDragDrop:=AValue;
end;

procedure TDragDropablePicture.BorderCollision;

begin
  if Top < Border.Top.y then begin
    Top := Border.Top.y;
  end;

  if IntersectsWith(Border.Bottom) then begin
    Top := Border.Bottom.y - Height - 1;
  end;

  if Left < Border.Left.x then begin
    Left := Border.Left.x;
  end;

  if IntersectsWith(Border.Right) then begin
    Left := Border.Right.x - Width - 1;
  end;
end;

procedure TDragDropablePicture.DragDrop(Sender, Source: TObject; X, Y: Integer);
begin

end;

procedure TDragDropablePicture.MouseDown(Sender: TObject;
  Shift: TCustomShiftState; X, Y: Integer);
begin
  if FDraggable then begin
    if (not FIsDragging) and (not SomeInstanceIsDragging) then begin
      //if ssLeft in Shift then begin
        FOffSet.X := X - Left;
        FOffSet.Y := Y - Top;
        FIsDragging := True;
        SomeInstanceIsDragging := True;
      //end;
    end;
  end;
  inherited MouseDown(Self, Shift, X, Y);
end;

procedure TDragDropablePicture.MouseMove(Sender: TObject;
  Shift: TCustomShiftState; X, Y: Integer);
begin
  if FDraggable then begin
    if FIsDragging and SomeInstanceIsDragging then begin
      Left := X - FOffSet.X;
      Top  := Y - FOffSet.Y;
      BorderCollision;
      Timestamp(ClassName+'.Move'+'.'+FCustomName, ToJSON);
    end;
  end;
  inherited MouseMove(Self, Shift, X, Y);
end;

procedure TDragDropablePicture.MouseUp(Sender: TObject;
  Shift: TCustomShiftState; X, Y: Integer);
var
  i : integer;
  LIntersectedPicture : TDragDropablePicture;
  LIntersected : Boolean;
begin
  if FDraggable then begin
    if FIsDragging and SomeInstanceIsDragging then begin
    //if ssLeft in Shift then begin
      FIsDragging := False;
      SomeInstanceIsDragging := False;
      for i := 0 to Choices.Count -1 do begin
        if Choices[i] is TDragDropablePicture then begin
          LIntersectedPicture := Choices[i] as TDragDropablePicture;
          LIntersected := IntersectsWith(LIntersectedPicture.BoundsRect);
          if LIntersected then Break;
        end;
      end;

      if LIntersectedPicture.Visible then begin
        if LIntersected then begin
          case i of
            0 :
              if Assigned(OnRightDragDrop) then
                OnRightDragDrop(LIntersectedPicture, Self, X, Y);

            else
              if Assigned(OnWrongDragDrop) then
                OnWrongDragDrop(LIntersectedPicture, Self, X, Y);
          end
        end else begin
          if Assigned(OnOtherDragDrop) then
            OnOtherDragDrop(nil, Self, X, Y);
        end;
      end;
    end;
  end;
  inherited MouseUp(Sender, Shift, X, Y);
end;

procedure TDragDropablePicture.MouseEnter(Sender: TObject);
begin
  inherited MouseEnter(Self);
end;

procedure TDragDropablePicture.MouseExit(Sender: TObject);
begin
  inherited MouseExit(Self);
end;

procedure TDragDropablePicture.Paint;
begin
  inherited Paint;
end;

constructor TDragDropablePicture.Create;
begin
  inherited Create;
  FBorder := Border;
  FBresenhamLine := TBresenhamLine.Create;
end;

destructor TDragDropablePicture.Destroy;
begin
  FBresenhamLine.Free;
  inherited Destroy;
end;

procedure TDragDropablePicture.Load(AParameters: TStringList; AParent: TObject;
  ARect: TSDL_Rect);
begin

end;

procedure TDragDropablePicture.Start;
begin
  Show;
end;

procedure TDragDropablePicture.Stop;
begin
  Hide;
end;

procedure TDragDropablePicture.DoResponse(AHuman: Boolean);
begin

end;

procedure TDragDropablePicture.UpdateDistance;
var
  LTarget : TDragDropablePicture;
begin
  if Assigned(TargetChoice) then begin
    LTarget := TargetChoice as TDragDropablePicture;
    FBresenhamLine.Update(BoundsRect, LTarget.BoundsRect);
  end;
end;

procedure TDragDropablePicture.MoveToPoint(APorcentage: Float);
var
  Point : TPoint;
begin
  if Assigned(TargetChoice) then begin
    Point := FBresenhamLine.GetPoint(APorcentage);
    Left := Point.X;
    Top := Point.Y;
    SetOriginalBounds;
  end;
end;

function TDragDropablePicture.GetID: TStimulusID;
begin
  Result.IsSample := FIsSample;
  Result.SubjcID := Pool.Counters.Subject;
  Result.SessiID := Pool.Session.ID;
  Result.BlockID := Pool.Session.Block.UID;
  Result.TrialID := Pool.Session.Trial.UID;
  //Result.StimuID := FIndex;
  //Result.RespoID := FResponseID;
  Result.Name := FCustomName;
end;

function TDragDropablePicture.ToData: string;
begin
  Result := FCustomName+'-'+FPosition.ToString;
end;

function TDragDropablePicture.ToJSON: string;
begin
  Result := '{'+String.Join(',', [
    'name:'          + FCustomName,
    'rect:'          + BoundsRect.ToJSON])+'}';
end;


end.

