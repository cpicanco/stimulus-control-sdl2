{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimulus;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, SDL2
  , sdl.app.graphics.rectangule
  , sdl.app.stimulus.contract
  , sdl.app.selectable.list
  , sdl.app.controls.custom
  , sdl.app.events.abstract
  , sdl.app.stimulus.types
  , session.loggers.types
  ;

type

  TModality = (ModalityNone, ModalityA, ModalityB, ModalityC, ModalityD);

  { TStimulus }

  TStimulus = class(TInterfacedObject, IStimulus)
    private
      FAnimated: Boolean;
      FModality: TModality;
      FName: string;
      FOnNoResponse: TNotifyEvent;
      FPosition: Integer;
      FResponseID : Integer;
      FSibling: TStimulus;
      FStimuli: TObject;
      FIndex : integer;
      FIsSample: Boolean;
      FOnMouseDown: TOnMouseEvent;
      FOnMouseEnter: TNotifyEvent;
      FOnMouseExit: TNotifyEvent;
      FOnMouseMove: TOnMouseEvent;
      FOnMouseUp: TOnMouseEvent;
      FOnResponse: TNotifyEvent;
      procedure SetIsSample(AValue: Boolean);
      procedure SetOnMouseDown(AValue: TOnMouseEvent);
      procedure SetOnMouseEnter(AValue: TNotifyEvent);
      procedure SetOnMouseExit(AValue: TNotifyEvent);
      procedure SetOnMouseMove(AValue: TOnMouseEvent);
      procedure SetOnMouseUp(AValue: TOnMouseEvent);
      procedure SetOnNoResponse(AValue: TNotifyEvent);
      procedure SetOnResponse(AValue: TNotifyEvent);
      procedure SetStimuli(AValue: TObject);
    protected
      FCustomName : string; { Filename only without extention }
      FSelectables : TSelectables;
      function GetID : TStimulusID;
      function ToData: string;
      function GetRect: TRectangule; virtual; abstract;
      function GetStimulusName : string; virtual; abstract;
      procedure DoResponseIncrement;
      procedure SetRect(AValue: TRectangule); virtual; abstract;
      procedure SetSibling(AValue: TStimulus); virtual;
      procedure MouseDown(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer); virtual; abstract;
      procedure MouseUp(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer); virtual; abstract;
      procedure MouseEnter(Sender:TObject); virtual; abstract;
      procedure MouseExit(Sender:TObject); virtual; abstract;
      procedure GazeEnter(Sender:TObject); virtual; abstract;
      procedure GazeExit(Sender:TObject); virtual; abstract;
    public
      constructor Create; virtual;
      destructor Destroy; override;
      function AsInterface : IStimulus;
      function IsCorrectResponse : Boolean; virtual; abstract;
      procedure Load(AParameters : TStringList;
        AParent : TObject; ARect: TSDL_Rect); virtual; abstract;
      procedure DoResponse(AHuman : Boolean); virtual;
      procedure Start; virtual; abstract;
      procedure Stop; virtual; abstract;
      procedure Hide; virtual;
      property OnNoResponse : TNotifyEvent read FOnNoResponse write SetOnNoResponse;
      property OnMouseMove: TOnMouseEvent read FOnMouseMove write SetOnMouseMove;
      property OnMouseDown: TOnMouseEvent read FOnMouseDown write SetOnMouseDown;
      property OnMouseUp: TOnMouseEvent read FOnMouseUp write SetOnMouseUp;
      property OnMouseEnter: TNotifyEvent read FOnMouseEnter write SetOnMouseEnter;
      property OnMouseExit: TNotifyEvent read FOnMouseExit write SetOnMouseExit;
      property OnResponse : TNotifyEvent read FOnResponse write SetOnResponse;
      property IsSample : Boolean read FIsSample write SetIsSample;
      property Selectables : TSelectables read FSelectables;
      property Index : Integer read FIndex write FIndex;
      property Name : string read FName write FName;
      property Modality : TModality read FModality write FModality;
      property Position : Integer read FPosition write FPosition;
      property Rectangule  : TRectangule read GetRect write SetRect;
      property ResponseID : integer read FResponseID;
      property Stimuli : TObject read FStimuli write SetStimuli;
      property Animated : Boolean read FAnimated write FAnimated;
      property Sibling : TStimulus read FSibling write FSibling;
  end;


implementation

uses session.pool, session.loggers.writerow.timestamp;

{ TStimulus }

procedure TStimulus.SetIsSample(AValue: Boolean);
begin
  if FIsSample=AValue then Exit;
  FIsSample:=AValue;
end;

function TStimulus.GetID: TStimulusID;
begin
  Result.IsSample := FIsSample;
  Result.SubjcID := Pool.Counters.Subject;
  Result.SessiID := Pool.Session.ID;
  Result.BlockID := Pool.Session.Block.UID;
  Result.TrialID := Pool.Session.Trial.UID;
  Result.StimuID := FIndex;
  Result.RespoID := FResponseID;
  Result.Name := GetStimulusName;
end;

procedure TStimulus.SetOnMouseDown(AValue: TOnMouseEvent);
begin
  if FOnMouseDown=AValue then Exit;
  FOnMouseDown:=AValue;
end;

procedure TStimulus.SetOnMouseEnter(AValue: TNotifyEvent);
begin
  if FOnMouseEnter=AValue then Exit;
  FOnMouseEnter:=AValue;
end;

procedure TStimulus.SetOnMouseExit(AValue: TNotifyEvent);
begin
  if FOnMouseExit=AValue then Exit;
  FOnMouseExit:=AValue;
end;

procedure TStimulus.SetOnMouseMove(AValue: TOnMouseEvent);
begin
  if FOnMouseMove=AValue then Exit;
  FOnMouseMove:=AValue;
end;

procedure TStimulus.SetOnMouseUp(AValue: TOnMouseEvent);
begin
  if FOnMouseUp=AValue then Exit;
  FOnMouseUp:=AValue;
end;

procedure TStimulus.SetOnNoResponse(AValue: TNotifyEvent);
begin
  if FOnNoResponse = AValue then Exit;
  FOnNoResponse := AValue;
end;

procedure TStimulus.SetOnResponse(AValue: TNotifyEvent);
begin
  if FOnResponse=AValue then Exit;
  FOnResponse:=AValue;
end;

procedure TStimulus.SetSibling(AValue: TStimulus);
begin
  if FSibling = AValue then Exit;
  FSibling := AValue;
end;

procedure TStimulus.SetStimuli(AValue: TObject);
begin
  if FStimuli = AValue then Exit;
  FStimuli := AValue;
end;

function TStimulus.ToData: string;
begin
  Result := FCustomName+'-'+FPosition.ToString;
end;

procedure TStimulus.DoResponseIncrement;
begin
 Inc(FResponseID);
end;

constructor TStimulus.Create;
begin
  FResponseID := 0;
  FSelectables := TSelectables.Create;
end;

destructor TStimulus.Destroy;
begin
  FOnMouseDown := nil;
  FOnMouseEnter := nil;
  FOnMouseExit := nil;
  FOnMouseMove := nil;
  FOnMouseUp := nil;
  FOnResponse := nil;
  FSelectables.Free;
  inherited Destroy;
end;

function TStimulus.AsInterface: IStimulus;
begin
  Result := Self as IStimulus;
end;

procedure TStimulus.DoResponse(AHuman: Boolean);
begin
  Inc(FResponseID);
  if AHuman then begin
    Timestamp('Stimulus.Response.'+GetID.ToString);
  end else begin
    Timestamp('Stimulus.Robot.Response.'+GetID.ToString);
  end;

  if Assigned(OnResponse) then
    OnResponse(Self);
end;

procedure TStimulus.Hide;
begin

end;

end.

