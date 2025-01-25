{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.controls.custom;

{$mode ObjFPC}{$H+}

interface

uses Classes, SysUtils, SDL2, Generics.Collections
  , session.parameters
  , sdl.app.paintable.contract
  , sdl.app.clickable.contract
  , sdl.app.moveable.contract
  , sdl.app.lookable.contract
  , sdl.app.selectable.contract
  , sdl.app.events.abstract
  ;

type

  TOnEyeFixationEvent = procedure(Sender: TObject) of object;
  TOnEyeSaccadeEvent = procedure(Sender: TObject) of object;
  TOnMouseEvent = procedure(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer) of object;

  TCallbacks = record
    OnResponse : TNotifyEvent;
    OnMouseMove: TOnMouseEvent;
    OnMouseDown: TOnMouseEvent;
    OnMouseUp: TOnMouseEvent;
    OnMouseEnter: TNotifyEvent;
    OnMouseExit: TNotifyEvent;
    OnNoResponse: TNotifyEvent;
  end;

  TChildren = specialize TList<TObject>;

  { TSDLControl }

  TSDLControl = class(TParametricObject,
    IClickable, IPaintable, IMoveable, ILookable, ISelectable)
  private
    FOnGazeEnter: TNotifyEvent;
    FOnGazeExit: TNotifyEvent;
    FOnGazeMove: TOnMouseEvent;
    FOwner: TObject;
    FGazeInside : Boolean;
    FMouseInside : Boolean;
    FOnMouseMove: TOnMouseEvent;
    FOnMouseDown: TOnMouseEvent;
    FOnMouseUp: TOnMouseEvent;
    FOnMouseEnter: TNotifyEvent;
    FOnMouseExit: TNotifyEvent;
    FParent : TSDLControl;
    function GetCustomName : string;
    function GetInFront: Boolean;
    function GetSDLMouseMotion    : TOnMouseMotionEvent;
    function GetSDLMouseButtonDown: TOnMouseButtonDownEvent;
    function GetSDLMouseButtonUp  : TOnMouseButtonUpEvent;
    function GetZIndex: integer;
    procedure SetCustomName(AValue : string);
    procedure SetOnGazeEnter(AValue: TNotifyEvent);
    procedure SetOnGazeExit(AValue: TNotifyEvent);
    procedure SetOnGazeMove(AValue: TOnMouseEvent);
    procedure SetOnMouseDown(AValue: TOnMouseEvent);
    procedure SetOnMouseEnter(AValue: TNotifyEvent);
    procedure SetOnMouseExit(AValue: TNotifyEvent);
    procedure SetOnMouseMove(AValue: TOnMouseEvent);
    procedure SetOnMouseUp(AValue: TOnMouseEvent);
    procedure SetOwner(AValue: TObject);
    procedure SetParent(AParent: TSDLControl);
    procedure AddChild(AChild: TObject);
    procedure SDLMouseMotion(const event: TSDL_MouseMotionEvent);
    procedure SDLMouseButtonDown(const event: TSDL_MouseButtonEvent);
    procedure SDLMouseButtonUp(const event: TSDL_MouseButtonEvent);
    procedure BringChildToFront(AChild : TObject);
  protected
    FInvalidated : Boolean;
    FCustomName : string;
    FRect : TSDL_Rect;
    FChildren : TChildren;
    function GetGazeInside : Boolean; virtual;
    function GetMouseInside : Boolean; virtual;
    function PointInside(SDLPoint : TSDL_Point) : Boolean;
    function IsGazeInside(SDLPoint : TSDL_Point;
      AInflateFactor: UInt32 = 0) : Boolean;
    function GetBoundsRect : TSDL_Rect; virtual;
    procedure SetBoundsRect(AValue : TSDL_Rect); virtual;
    procedure SetMouseInside(AValue : Boolean);
    procedure SetGazeInside(AValue : Boolean);
    procedure Paint; virtual; abstract;
    procedure GazeEnter(Sender: TObject); virtual;
    procedure GazeExit(Sender: TObject); virtual;
    procedure GazeMove(Sender: TObject;
      Shift: TCustomShiftState; X,Y: Integer); virtual;
    procedure MouseMove(Sender: TObject;
      Shift: TCustomShiftState; X,Y: Integer); virtual;
    procedure MouseDown(Sender: TObject;
      Shift: TCustomShiftState; X, Y: Integer); virtual;
    procedure MouseUp(Sender: TObject;
      Shift: TCustomShiftState; X, Y: Integer); virtual;
    procedure MouseEnter(Sender: TObject); virtual;
    procedure MouseExit(Sender: TObject); virtual;
  public
    constructor Create; virtual;
    destructor Destroy; override;
    function InflateBoundsRect(AFactor: UInt32) : TSDL_Rect;
    function AsIClickable : IClickable;
    function AsIPaintable : IPaintable;
    function AsIMoveable : IMoveable;
    function AsILookable : ILookable;
    function AsISelectable : ISelectable;
    function CenterPoint : TSDL_Point;
    function BottomRightPoint : TSDL_Point;
    function ClientToParent(APoint : TSDL_Point) : TSDL_Point;
    //function Invalidated : Boolean;
    function Origen : TSDL_Point;
    //procedure Invalidate;
    //procedure Validate;
    procedure BringToFront;
    procedure Confirm; virtual;
    procedure Select; virtual;
    property Parent : TSDLControl read FParent write SetParent;
    property BoundsRect : TSDL_Rect read GetBoundsRect;
    property OnGazeEnter : TNotifyEvent read FOnGazeEnter write SetOnGazeEnter;
    property OnGazeExit : TNotifyEvent read FOnGazeExit write SetOnGazeExit;
    property OnGazeMove : TOnMouseEvent read FOnGazeMove write SetOnGazeMove;
    property OnMouseMove : TOnMouseEvent read FOnMouseMove write SetOnMouseMove;
    property OnMouseDown : TOnMouseEvent read FOnMouseDown write SetOnMouseDown;
    property OnMouseUp   : TOnMouseEvent read FOnMouseUp write SetOnMouseUp;
    property OnMouseEnter : TNotifyEvent read FOnMouseEnter write SetOnMouseEnter;
    property OnMouseExit : TNotifyEvent read FOnMouseExit write SetOnMouseExit;
    property MouseInside : Boolean read GetMouseInside write SetMouseInside;
    property ZIndex : integer read GetZIndex;
    property InFront : Boolean read GetInFront;
    property Owner : TObject read FOwner write SetOwner;
    property CustomName : string read GetCustomName write SetCustomName;
  end;

implementation

uses sdl.app.mouse, sdl.app.trials.factory;

{ TSDLControl }

procedure TSDLControl.SetParent(AParent: TSDLControl);
begin
  if FParent=AParent then Exit;
  FParent:=AParent;
  if FParent <> nil then
    FParent.AddChild(Self);
end;

procedure TSDLControl.SetOnMouseEnter(AValue: TNotifyEvent);
begin
  if FOnMouseEnter=AValue then Exit;
  FOnMouseEnter:=AValue;
end;

procedure TSDLControl.SetOnMouseExit(AValue: TNotifyEvent);
begin
  if FOnMouseExit=AValue then Exit;
  FOnMouseExit:=AValue;
end;
                                        
procedure TSDLControl.SetOnMouseMove(AValue: TOnMouseEvent);
begin
  if FOnMouseMove=AValue then Exit;
  FOnMouseMove:=AValue;
end;
                                        
procedure TSDLControl.SetOnMouseDown(AValue: TOnMouseEvent);
begin
  if FOnMouseDown=AValue then Exit;
  FOnMouseDown:=AValue;
end;

procedure TSDLControl.SetOnMouseUp(AValue: TOnMouseEvent);
begin
  if FOnMouseUp=AValue then Exit;
  FOnMouseUp:=AValue;
end;

procedure TSDLControl.SetOwner(AValue: TObject);
begin
  if FOwner = AValue then Exit;
  FOwner := AValue;
end;

function TSDLControl.GetSDLMouseMotion: TOnMouseMotionEvent;
begin
  Result := @SDLMouseMotion;
end;

function TSDLControl.GetInFront: Boolean;
begin
  Result := ZIndex = FParent.FChildren.Count-1;
end;

function TSDLControl.GetSDLMouseButtonDown: TOnMouseButtonDownEvent;
begin
  Result := @SDLMouseButtonDown;
end;

function TSDLControl.GetSDLMouseButtonUp: TOnMouseButtonUpEvent;
begin
  Result := @SDLMouseButtonUp;
end;

function TSDLControl.GetZIndex: integer;
begin
  Result := FParent.FChildren.IndexOf(Self);
end;

procedure TSDLControl.SetOnGazeEnter(AValue: TNotifyEvent);
begin
  if FOnGazeEnter = AValue then Exit;
  FOnGazeEnter := AValue;
end;

procedure TSDLControl.SetOnGazeExit(AValue: TNotifyEvent);
begin
  if FOnGazeExit = AValue then Exit;
  FOnGazeExit := AValue;
end;

procedure TSDLControl.SetOnGazeMove(AValue: TOnMouseEvent);
begin
  if FOnGazeMove = AValue then Exit;
  FOnGazeMove := AValue;
end;

procedure TSDLControl.AddChild(AChild: TObject);
begin
  FChildren.Add(AChild);
end;

function TSDLControl.GetBoundsRect: TSDL_Rect;
begin
  Result := FRect;
end;

procedure TSDLControl.SetBoundsRect(AValue: TSDL_Rect);
begin
  FRect := AValue;
end;

procedure TSDLControl.SetMouseInside(AValue: Boolean);
begin
  if FMouseInside=AValue then Exit;
  FMouseInside:=AValue;
end;

procedure TSDLControl.SetGazeInside(AValue: Boolean);
begin
  if FGazeInside=AValue then Exit;
  FGazeInside:=AValue;
end;

procedure TSDLControl.GazeEnter(Sender: TObject);
begin
  if Assigned(OnGazeEnter) then
    OnGazeEnter(Sender);
end;

procedure TSDLControl.GazeExit(Sender: TObject);
begin
  if Assigned(OnGazeExit) then
    OnGazeExit(Sender);
end;

procedure TSDLControl.GazeMove(Sender: TObject; Shift: TCustomShiftState;
  X, Y: Integer);
begin
  if Assigned(OnGazeMove) then
    OnGazeMove(Sender, GetShiftState, X, Y);
end;

procedure TSDLControl.SDLMouseButtonDown(const event: TSDL_MouseButtonEvent);
begin
  MouseDown(Self, GetShiftState, event.x, event.y);
end;

procedure TSDLControl.SDLMouseButtonUp(const event: TSDL_MouseButtonEvent);
begin
  MouseUp(Self, GetShiftState, event.x, event.y);
end;

procedure TSDLControl.BringChildToFront(AChild: TObject);
begin
  if FChildren.Count <= 1 then Exit;
  with FChildren do Add(Extract(AChild));
end;

function TSDLControl.GetGazeInside: Boolean;
begin
  Result := FGazeInside;
end;

function TSDLControl.GetMouseInside: Boolean;
begin
  Result := FMouseInside;
end;

function TSDLControl.PointInside(SDLPoint: TSDL_Point): Boolean;
var
  SDLRect  : TSDL_Rect;
begin
  SDLRect := GetBoundsRect;
  Result := SDL_PointInRect(@SDLPoint, @SDLRect);
end;

function TSDLControl.IsGazeInside(SDLPoint: TSDL_Point;
  AInflateFactor: UInt32): Boolean;
var
  SDLRect  : TSDL_Rect;
begin
  if AInflateFactor > 0 then begin
    SDLRect := InflateBoundsRect(AInflateFactor);
  end else begin
    SDLRect := GetBoundsRect;
  end;
  Result := SDL_PointInRect(@SDLPoint, @SDLRect);
end;

procedure TSDLControl.SDLMouseMotion(const event: TSDL_MouseMotionEvent);
begin
  MouseMove(Self, GetShiftState, event.x, event.y);
end;

procedure TSDLControl.MouseMove(Sender: TObject; Shift: TCustomShiftState;
  X, Y: Integer);
begin
  if Assigned(OnMouseMove) then
    OnMouseMove(Sender, Shift, X, Y);
end;

procedure TSDLControl.MouseDown(Sender: TObject; Shift: TCustomShiftState;
  X, Y: Integer);
begin
  if Assigned(OnMouseDown) then
    OnMouseDown(Sender, Shift, X, Y);
end;

procedure TSDLControl.MouseUp(Sender: TObject; Shift: TCustomShiftState;
  X, Y: Integer);
begin
  if Assigned(OnMouseUp) then
    OnMouseUp(Sender, Shift, X, Y);
end;

procedure TSDLControl.MouseEnter(Sender: TObject);
begin
  if Assigned(OnMouseEnter) then
    OnMouseEnter(Sender);
end;

procedure TSDLControl.MouseExit(Sender: TObject);
begin
  if Assigned(OnMouseExit) then
    OnMouseExit(Sender);
end;

constructor TSDLControl.Create;
begin
  inherited Create;
  FInvalidated := False;
  FCustomName := '';
  FChildren := TChildren.Create;
  FMouseInside := False;
  FGazeInside := False;
  FParent := nil;
end;

destructor TSDLControl.Destroy;
begin
  FChildren.Free;
  inherited Destroy;
end;

function TSDLControl.InflateBoundsRect(AFactor: UInt32): TSDL_Rect;
begin
  Result.x := FRect.x - AFactor;
  Result.y := FRect.y - AFactor;
  Result.w := FRect.w + (2 * AFactor);
  Result.h := FRect.h + (2 * AFactor);
end;

function TSDLControl.AsIClickable: IClickable;
begin
  Result := Self as IClickable;
end;

function TSDLControl.AsIPaintable: IPaintable;
begin
  Result := Self as IPaintable;
end;

function TSDLControl.AsIMoveable: IMoveable;
begin
  Result := Self as IMoveable;
end;

function TSDLControl.AsILookable: ILookable;
begin
  Result := Self as ILookable;
end;

function TSDLControl.AsISelectable: ISelectable;
begin
  Result := Self as ISelectable;
end;

function TSDLControl.CenterPoint: TSDL_Point;
begin
  Result.X := FRect.w div 2;
  Result.Y := FRect.h div 2;
end;

function TSDLControl.GetCustomName: string;
begin
  Result := FCustomName;
end;

function TSDLControl.BottomRightPoint: TSDL_Point;
begin
  Result.X := FRect.w;
  Result.Y := FRect.h;
end;

function TSDLControl.ClientToParent(APoint: TSDL_Point): TSDL_Point;
begin
  Result.X := APoint.X + FRect.x;
  Result.Y := APoint.y + FRect.y;
end;

//function TSDLControl.Invalidated: Boolean;
//begin
//  Result := FInvalidated;
//end;

function TSDLControl.Origen: TSDL_Point;
begin
  Result.X := FRect.x + (FRect.w div 2);
  Result.Y := FRect.y + (FRect.h div 2);
end;

//procedure TSDLControl.Invalidate;
//begin
//  FInvalidated := True;
//end;
//
//procedure TSDLControl.Validate;
//begin
//  FInvalidated := False;
//end;

procedure TSDLControl.BringToFront;
begin
  FParent.BringChildToFront(Self);
end;

procedure TSDLControl.Confirm;
var
  LPoint : TSDL_Point;
begin
  Mouse.State(LPoint);
  if PointInside(LPoint) then begin
    MouseUp(Self, GetShiftState, LPoint.X, LPoint.Y);
  end;
end;

procedure TSDLControl.Select;
begin
  Mouse.MoveTo(ClientToParent(CenterPoint));
end;

procedure TSDLControl.SetCustomName(AValue: string);
begin
  if FCustomName = AValue then Exit;
  FCustomName := AValue;
end;

end.

